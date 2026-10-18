# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""
Tests for saturation handling in simulation and generated code.

ap_fixed<16,8> representable range:
  max =  127.99609375  (= 32767/256)
  min = -128.0         (= -32768/256)

Ramp input: element i of a [1,256] tensor has value i/256.

Model boundaries (saturating elements):
  sat_add_pos:  i >= 128  → i/256 + 127.5 >= 128.0        → saturate to max
  sat_add_neg:  i <= 127  → i/256 - 128.5 <= -128.004      → saturate to min
  sat_mul_pos:  i >= 109  → (i/256 + 1) * 90 >= 128.320    → saturate to max
  sat_mul_neg:  i <= 147  → (i/256 - 2) * 90 <= -128.320   → saturate to min
"""

import unittest

from helpers import _model, _models_exist
from src.graph   import OnnxGraph
from src.codegen import CodeGenerator

AP_MAX =  127.99609375   # 32767 / 256
AP_MIN = -128.0          #-32768 / 256


def _sim(model_name: str):
    """Return (CodeGenerator, simulated output array) for a saturation model."""
    g   = OnnxGraph(_model(model_name))
    cg  = CodeGenerator(g, _model(model_name))
    out = g.output_tensors[0].onnx_name
    return cg, cg._simulate()[out].flatten()


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatAddPos(unittest.TestCase):
    """X[1,256] + 127.5 -> Y.  Elements [128..255] saturate to max."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_add_pos.onnx")

    # ---------------------------------------------------------------- #
    # Structural checks                                                  #
    # ---------------------------------------------------------------- #

    def test_single_add_node(self):
        g = OnnxGraph(_model("sat_add_pos.onnx"))
        self.assertEqual(len(g.nodes), 1)

    def test_add_op_in_source(self):
        s = self._cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)

    # ---------------------------------------------------------------- #
    # Simulation values — non-saturating range                          #
    # ---------------------------------------------------------------- #

    def test_first_element_no_saturation(self):
        # Y[0] = 0/256 + 127.5 = 127.5 (exact, no saturation)
        self.assertAlmostEqual(self._Y[0], 127.5, places=6)

    def test_boundary_element_at_max(self):
        # Y[127] = 127/256 + 127.5 = 127.99609375 = AP_MAX (exact, no overflow)
        self.assertAlmostEqual(self._Y[127], AP_MAX, places=6)

    # ---------------------------------------------------------------- #
    # Simulation values — saturating range                              #
    # ---------------------------------------------------------------- #

    def test_first_saturating_element(self):
        # Y[128] = 128/256 + 127.5 = 128.0 → saturate to AP_MAX
        self.assertAlmostEqual(self._Y[128], AP_MAX, places=6)

    def test_last_element_saturated(self):
        self.assertAlmostEqual(self._Y[255], AP_MAX, places=6)

    def test_saturated_count(self):
        import numpy as np
        # Elements 127..255 all equal AP_MAX (127 exactly at max, 128..255 clamped)
        count = int(np.sum(self._Y == AP_MAX))
        self.assertEqual(count, 129)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatAddNeg(unittest.TestCase):
    """X[1,256] + (-63) + (-65.5) -> Y.  Elements [0..127] saturate to min."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_add_neg.onnx")

    def test_two_add_nodes(self):
        g = OnnxGraph(_model("sat_add_neg.onnx"))
        self.assertEqual(len(g.nodes), 2)

    def test_first_element_saturated(self):
        # Y[0] = 0 - 128.5 = -128.5 → saturate to AP_MIN
        self.assertAlmostEqual(self._Y[0], AP_MIN, places=6)

    def test_last_saturating_element(self):
        # Y[127] = 127/256 - 128.5 = -128.004 → saturate to AP_MIN
        self.assertAlmostEqual(self._Y[127], AP_MIN, places=6)

    def test_boundary_at_min(self):
        # Y[128] = 128/256 - 128.5 = -128.0 = AP_MIN exactly (no saturation needed)
        self.assertAlmostEqual(self._Y[128], AP_MIN, places=6)

    def test_first_non_saturating_element(self):
        # Y[129] = 129/256 - 128.5 = 0.50390625 - 128.5 = -127.99609375
        expected = 129 / 256.0 - 128.5
        self.assertAlmostEqual(self._Y[129], expected, places=6)

    def test_last_element_no_saturation(self):
        # Y[255] = 255/256 - 128.5 = -127.50390625
        expected = 255 / 256.0 - 128.5
        self.assertAlmostEqual(self._Y[255], expected, places=6)

    def test_saturated_count(self):
        import numpy as np
        # Elements 0..128 all equal AP_MIN (0..127 clamped, 128 exactly at min)
        count = int(np.sum(self._Y == AP_MIN))
        self.assertEqual(count, 129)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatMulPos(unittest.TestCase):
    """(X + 1) * 90 -> Y.  Elements [109..255] saturate to max."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_mul_pos.onnx")

    def test_add_and_mul_nodes(self):
        g = OnnxGraph(_model("sat_mul_pos.onnx"))
        self.assertEqual(len(g.nodes), 2)

    def test_ops_in_source(self):
        s = self._cg.generate_source()
        self.assertIn("VECTOROP_ADD", s)
        self.assertIn("VECTOROP_MUL", s)

    def test_first_element_no_saturation(self):
        # Y[0] = (0/256 + 1) * 90 = 90.0 (exact)
        self.assertAlmostEqual(self._Y[0], 90.0, places=6)

    def test_last_non_saturating_element(self):
        # Z[108] = 364/256 = 1.421875 (exact on grid)
        # Y[108] = 364/256 * 23040/256 = 8386560/65536 = 32760/256 = 127.96875
        self.assertAlmostEqual(self._Y[108], 127.96875, places=6)

    def test_first_saturating_element(self):
        # Z[109] = 365/256, Z * 90 = 8409600/65536 = 128.32... → saturate
        self.assertAlmostEqual(self._Y[109], AP_MAX, places=6)

    def test_last_element_saturated(self):
        self.assertAlmostEqual(self._Y[255], AP_MAX, places=6)

    def test_saturated_count(self):
        import numpy as np
        count = int(np.sum(self._Y == AP_MAX))
        self.assertEqual(count, 147)   # indices 109..255


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatMulNeg(unittest.TestCase):
    """(X - 2) * 90 -> Y.  Elements [0..147] saturate to min."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_mul_neg.onnx")

    def test_sub_and_mul_nodes(self):
        g = OnnxGraph(_model("sat_mul_neg.onnx"))
        self.assertEqual(len(g.nodes), 2)

    def test_ops_in_source(self):
        s = self._cg.generate_source()
        self.assertIn("VECTOROP_SUB", s)
        self.assertIn("VECTOROP_MUL", s)

    def test_first_element_saturated(self):
        # Y[0] = (0 - 2) * 90 = -180 → saturate to AP_MIN
        self.assertAlmostEqual(self._Y[0], AP_MIN, places=6)

    def test_last_saturating_element(self):
        # Z[147] = -365/256 = -1.42578125, Y = -128.32... → saturate
        self.assertAlmostEqual(self._Y[147], AP_MIN, places=6)

    def test_first_non_saturating_element(self):
        # Z[148] = -364/256 = -1.421875 (exact on grid)
        # Y[148] = -364/256 * 23040/256 = -8386560/65536 = -32760/256 = -127.96875
        self.assertAlmostEqual(self._Y[148], -127.96875, places=6)

    def test_last_element_no_saturation(self):
        # Z[255] = (255/256 - 2) = -265/256 = -1.03515625
        # Y[255] = -265/256 * 23040/256 = -6105600/65536 = floor(-23850.0)/256 = -23130/256
        # Let simulation determine; just check not saturated
        self.assertGreater(self._Y[255], AP_MIN)

    def test_saturated_count(self):
        import numpy as np
        count = int(np.sum(self._Y == AP_MIN))
        self.assertEqual(count, 148)   # indices 0..147


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatSubPos(unittest.TestCase):
    """X[1,256] - (-127.5) -> Y.  Elements [128..255] saturate to max."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_sub_pos.onnx")

    def test_single_sub_node(self):
        g = OnnxGraph(_model("sat_sub_pos.onnx"))
        self.assertEqual(len(g.nodes), 1)

    def test_sub_op_in_source(self):
        s = self._cg.generate_source()
        self.assertIn("VECTOROP_SUB", s)

    def test_first_element_no_saturation(self):
        # Y[0] = 0/256 - (-127.5) = 127.5 (exact, no saturation)
        self.assertAlmostEqual(self._Y[0], 127.5, places=6)

    def test_boundary_element_at_max(self):
        # Y[127] = 127/256 + 127.5 = 127.99609375 = AP_MAX (exact, no overflow)
        self.assertAlmostEqual(self._Y[127], AP_MAX, places=6)

    def test_first_saturating_element(self):
        # Y[128] = 128/256 + 127.5 = 128.0 → saturate to AP_MAX
        self.assertAlmostEqual(self._Y[128], AP_MAX, places=6)

    def test_last_element_saturated(self):
        self.assertAlmostEqual(self._Y[255], AP_MAX, places=6)

    def test_saturated_count(self):
        import numpy as np
        # Elements 127..255 all equal AP_MAX (127 exactly at max, 128..255 clamped)
        count = int(np.sum(self._Y == AP_MAX))
        self.assertEqual(count, 129)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatSubNeg(unittest.TestCase):
    """X[1,256] - 63.0 - 65.5 -> Y.  Elements [0..128] saturate to min."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_sub_neg.onnx")

    def test_two_sub_nodes(self):
        g = OnnxGraph(_model("sat_sub_neg.onnx"))
        self.assertEqual(len(g.nodes), 2)

    def test_sub_op_in_source(self):
        s = self._cg.generate_source()
        self.assertIn("VECTOROP_SUB", s)

    def test_first_element_saturated(self):
        # Y[0] = 0 - 128.5 = -128.5 → saturate to AP_MIN
        self.assertAlmostEqual(self._Y[0], AP_MIN, places=6)

    def test_last_saturating_element(self):
        # Y[127] = 127/256 - 128.5 = -128.004 → saturate to AP_MIN
        self.assertAlmostEqual(self._Y[127], AP_MIN, places=6)

    def test_boundary_at_min(self):
        # Y[128] = 128/256 - 128.5 = -128.0 = AP_MIN exactly (no saturation needed)
        self.assertAlmostEqual(self._Y[128], AP_MIN, places=6)

    def test_first_non_saturating_element(self):
        # Y[129] = 129/256 - 128.5 = -127.99609375
        expected = 129 / 256.0 - 128.5
        self.assertAlmostEqual(self._Y[129], expected, places=6)

    def test_last_element_no_saturation(self):
        # Y[255] = 255/256 - 128.5 = -127.50390625
        expected = 255 / 256.0 - 128.5
        self.assertAlmostEqual(self._Y[255], expected, places=6)

    def test_saturated_count(self):
        import numpy as np
        # Elements 0..128 all equal AP_MIN (0..127 clamped, 128 exactly at min)
        count = int(np.sum(self._Y == AP_MIN))
        self.assertEqual(count, 129)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatDivPos(unittest.TestCase):
    """X[1,256] / (1/256) -> Y = float(i).  Elements [128..255] saturate to max."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_div_pos.onnx")

    def test_single_div_node(self):
        g = OnnxGraph(_model("sat_div_pos.onnx"))
        self.assertEqual(len(g.nodes), 1)

    def test_div_op_in_source(self):
        s = self._cg.generate_source()
        self.assertIn("VECTOROP_DIV", s)

    def test_first_element_no_saturation(self):
        # Y[0] = (0/256) / (1/256) = 0.0
        self.assertAlmostEqual(self._Y[0], 0.0, places=6)

    def test_last_non_saturating_element(self):
        # Y[127] = 127.0 (< AP_MAX = 127.996, no saturation)
        self.assertAlmostEqual(self._Y[127], 127.0, places=6)

    def test_first_saturating_element(self):
        # Y[128] = 128.0 > AP_MAX → saturate to AP_MAX
        self.assertAlmostEqual(self._Y[128], AP_MAX, places=6)

    def test_last_element_saturated(self):
        self.assertAlmostEqual(self._Y[255], AP_MAX, places=6)

    def test_saturated_count(self):
        import numpy as np
        # Elements 128..255 all equal AP_MAX
        count = int(np.sum(self._Y == AP_MAX))
        self.assertEqual(count, 128)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatDivNeg(unittest.TestCase):
    """X[1,256] / (-1/256) -> Y = float(-i).  Elements [128..255] saturate to min."""

    def setUp(self):
        self._cg, self._Y = _sim("sat_div_neg.onnx")

    def test_single_div_node(self):
        g = OnnxGraph(_model("sat_div_neg.onnx"))
        self.assertEqual(len(g.nodes), 1)

    def test_div_op_in_source(self):
        s = self._cg.generate_source()
        self.assertIn("VECTOROP_DIV", s)

    def test_first_element_no_saturation(self):
        # Y[0] = (0/256) / (-1/256) = 0.0
        self.assertAlmostEqual(self._Y[0], 0.0, places=6)

    def test_last_non_saturating_element(self):
        # Y[127] = -127.0 (> AP_MIN = -128.0, no saturation)
        self.assertAlmostEqual(self._Y[127], -127.0, places=6)

    def test_boundary_element_at_min(self):
        # Y[128] = -128.0 = AP_MIN exactly (representable, no overflow)
        self.assertAlmostEqual(self._Y[128], AP_MIN, places=6)

    def test_first_saturating_element(self):
        # Y[129] = -129.0 → saturate to AP_MIN
        self.assertAlmostEqual(self._Y[129], AP_MIN, places=6)

    def test_last_element_saturated(self):
        self.assertAlmostEqual(self._Y[255], AP_MIN, places=6)

    def test_saturated_count(self):
        import numpy as np
        # Elements 128..255 all equal AP_MIN (128 at exact boundary, 129..255 clamped)
        count = int(np.sum(self._Y == AP_MIN))
        self.assertEqual(count, 128)


@unittest.skipUnless(_models_exist(), "Run test/gen_test_models.py first")
class TestSatExpectedArrays(unittest.TestCase):
    """Verify the generated C expected arrays contain the saturated bit patterns."""

    SAT_MAX_PATTERN = "0x7FFF"   # 32767 = AP_MAX in ap_fixed<16,8> storage
    SAT_MIN_PATTERN = "0x8000"   # -32768 = AP_MIN (two's complement)

    def _test_src(self, model_name: str) -> str:
        g  = OnnxGraph(_model(model_name))
        cg = CodeGenerator(g, _model(model_name))
        return cg.generate_test()

    def test_sat_add_pos_max_in_expected(self):
        t = self._test_src("sat_add_pos.onnx")
        self.assertIn(self.SAT_MAX_PATTERN, t)

    def test_sat_add_neg_min_in_expected(self):
        t = self._test_src("sat_add_neg.onnx")
        self.assertIn(self.SAT_MIN_PATTERN, t)

    def test_sat_mul_pos_max_in_expected(self):
        t = self._test_src("sat_mul_pos.onnx")
        self.assertIn(self.SAT_MAX_PATTERN, t)

    def test_sat_mul_neg_min_in_expected(self):
        t = self._test_src("sat_mul_neg.onnx")
        self.assertIn(self.SAT_MIN_PATTERN, t)

    def test_sat_sub_pos_max_in_expected(self):
        t = self._test_src("sat_sub_pos.onnx")
        self.assertIn(self.SAT_MAX_PATTERN, t)

    def test_sat_sub_neg_min_in_expected(self):
        t = self._test_src("sat_sub_neg.onnx")
        self.assertIn(self.SAT_MIN_PATTERN, t)

    def test_sat_div_pos_max_in_expected(self):
        t = self._test_src("sat_div_pos.onnx")
        self.assertIn(self.SAT_MAX_PATTERN, t)

    def test_sat_div_neg_min_in_expected(self):
        t = self._test_src("sat_div_neg.onnx")
        self.assertIn(self.SAT_MIN_PATTERN, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)

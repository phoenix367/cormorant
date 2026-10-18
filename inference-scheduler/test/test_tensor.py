# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ap_fixed<16,8> weight encoding via DataType abstraction."""

import unittest
import numpy as np

from src.dtype import AP_FIXED_16_8


class TestTensorEncoding(unittest.TestCase):
    """ap_fixed<16,8> weight encoding via AP_FIXED_16_8.encode_weight()."""

    def test_zero(self):
        self.assertEqual(AP_FIXED_16_8.encode_weight(np.array([0.0])), ["0x0000"])

    def test_one(self):
        # 1.0 * 256 = 256 = 0x0100
        self.assertEqual(AP_FIXED_16_8.encode_weight(np.array([1.0])), ["0x0100"])

    def test_minus_one(self):
        # -1.0 * 256 = -256 = 0xFF00 as uint16
        self.assertEqual(AP_FIXED_16_8.encode_weight(np.array([-1.0])), ["0xFF00"])

    def test_half(self):
        # 0.5 * 256 = 128 = 0x0080
        self.assertEqual(AP_FIXED_16_8.encode_weight(np.array([0.5])), ["0x0080"])

    def test_saturation_positive(self):
        # Values above max saturate to the max representable value
        max_val = 127.99609375   # ap_fixed<16,8> max: 127 + 255/256
        encoded_saturated = AP_FIXED_16_8.encode_weight(np.array([999.0]))
        encoded_max       = AP_FIXED_16_8.encode_weight(np.array([max_val]))
        self.assertEqual(encoded_saturated, encoded_max)

    def test_saturation_negative(self):
        # Values below min saturate to the min representable value
        min_val = -128.0         # ap_fixed<16,8> min: -128
        encoded_saturated = AP_FIXED_16_8.encode_weight(np.array([-999.0]))
        encoded_min       = AP_FIXED_16_8.encode_weight(np.array([min_val]))
        self.assertEqual(encoded_saturated, encoded_min)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for src.dtype — Float32 path + DataType default implementations.

Most production code paths exercise ApFixed<16,8>, so Float32 and the
DataType-base default methods (format_literal, c_typedef_comment,
truncate, truncate_div) are otherwise unexercised.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from src.dtype import (
    DataType,
    ApFixed,
    Float32,
    AP_FIXED_16_8,
    FLOAT32,
)


# ================================================================ #
# DataType base-class default methods                               #
# ================================================================ #

class _MinimalDataType(DataType):
    """Implements only the abstract methods; inherits every default impl
    we want to exercise (format_literal, c_typedef_comment,
    align_elems)."""

    @property
    def name(self) -> str:           return "stub<8>"
    @property
    def bytes_per_elem(self) -> int: return 1
    @property
    def c_type(self) -> str:         return "uint8_t"
    @property
    def c_array_type(self) -> str:   return "uint8_t"
    @property
    def np_storage(self):            return np.uint8

    def quantize(self, x):           return x
    def ramp_to_float(self, p):      return p.astype(np.float64)
    def float_to_storage(self, x):   return x.astype(np.uint8)
    def dat_bytes(self, d):          return d.astype(np.uint8).tobytes()
    def c_display(self, ptr, idx):   return f"{ptr}[{idx}]"
    def c_fill_rhs(self, expr):      return expr


class TestDataTypeDefaultMethods(unittest.TestCase):
    def setUp(self):
        self.dt = _MinimalDataType()

    def test_default_format_literal_hex(self):
        # bytes_per_elem=1 → 2 hex digits, mask=0xFF.
        self.assertEqual(self.dt.format_literal(0xAB), "0xAB")
        # Negative inputs are masked to the storage width.
        self.assertEqual(self.dt.format_literal(-1), "0xFF")

    def test_default_c_typedef_comment(self):
        self.assertEqual(self.dt.c_typedef_comment(), "/* stub<8> */")

    def test_default_align_elems_derived_from_bytes(self):
        # Default impl: ALIGN_BYTES / bytes_per_elem.
        # For bytes_per_elem=1 and ALIGN_BYTES=16 → align_elems=16.
        self.assertEqual(self.dt.align_elems, 16)


class TestDataTypeDefaultTruncate(unittest.TestCase):
    """Float32 does not override truncate / truncate_div so the
    base-class default implementations are reached via FLOAT32."""

    def test_truncate_delegates_to_quantize(self):
        x = np.array([1.5, -2.5, 3.75], dtype=np.float64)
        np.testing.assert_array_equal(FLOAT32.truncate(x), FLOAT32.quantize(x))

    def test_truncate_div_delegates_to_truncate(self):
        x = np.array([1.5, -2.5, 3.75], dtype=np.float64)
        np.testing.assert_array_equal(
            FLOAT32.truncate_div(x), FLOAT32.truncate(x))


# ================================================================ #
# ApFixed constructor validation                                    #
# ================================================================ #

class TestApFixedConstructorErrors(unittest.TestCase):
    def test_unsupported_width_raises(self):
        with self.assertRaisesRegex(ValueError, "W=4 not supported"):
            ApFixed(4, 2)
        with self.assertRaisesRegex(ValueError, "W=64 not supported"):
            ApFixed(64, 32)

    def test_integer_bits_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "I=0 out of range"):
            ApFixed(16, 0)
        with self.assertRaisesRegex(ValueError, "I=16 out of range"):
            ApFixed(16, 16)

    def test_supported_widths_construct_cleanly(self):
        for W in (8, 16, 32):
            ApFixed(W, W // 2)  # any valid I works


# ================================================================ #
# Float32 — every method                                            #
# ================================================================ #

class TestFloat32Properties(unittest.TestCase):
    def test_name_and_storage_basics(self):
        self.assertEqual(FLOAT32.name, "float32")
        self.assertEqual(FLOAT32.bytes_per_elem, 4)
        self.assertEqual(FLOAT32.c_type, "float")
        self.assertEqual(FLOAT32.c_array_type, "float")
        self.assertEqual(FLOAT32.np_storage, np.float32)


class TestFloat32Numerics(unittest.TestCase):
    def test_quantize_round_trips_through_float32(self):
        # 1/3 isn't exactly representable in float32; round-trip introduces
        # the same precision loss as the hardware.
        x   = np.array([1.0 / 3.0, 1.0 / 7.0, 1.5], dtype=np.float64)
        out = FLOAT32.quantize(x)
        np.testing.assert_array_equal(out, x.astype(np.float32).astype(np.float64))

    def test_ramp_to_float_masks_to_16_bits(self):
        # Matches the C test harness: p[pos] = (float)(pos & 0xFFFFu).
        positions = np.array([0, 100, 0x10000, 0x1FFFF], dtype=np.int64)
        np.testing.assert_array_equal(
            FLOAT32.ramp_to_float(positions),
            (positions & 0xFFFF).astype(np.float64),
        )

    def test_float_to_storage_returns_float32_dtype(self):
        x   = np.array([1.5, -2.5, 100.0], dtype=np.float64)
        out = FLOAT32.float_to_storage(x)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, x, rtol=1e-6)


class TestFloat32Serialization(unittest.TestCase):
    def test_dat_bytes_little_endian(self):
        # 1.0 in little-endian IEEE 754 single = 0x3F800000 → bytes 00 00 80 3F
        data = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        buf  = FLOAT32.dat_bytes(data)
        self.assertEqual(len(buf), 12)  # 3 floats * 4 bytes
        self.assertEqual(buf[:4], b"\x00\x00\x80\x3F")     # 1.0
        self.assertEqual(buf[4:8], b"\x00\x00\x00\x40")    # 2.0
        self.assertEqual(buf[8:12], b"\x00\x00\x80\x40")   # 4.0


class TestFloat32Codegen(unittest.TestCase):
    def test_c_display_plain_cast(self):
        self.assertEqual(FLOAT32.c_display("buf", "i"), "(double)buf[i]")

    def test_c_fill_rhs_masks_to_16_bits(self):
        self.assertEqual(FLOAT32.c_fill_rhs("pos"), "(Data_t)(pos & 0xFFFFu)")

    def test_format_literal_emits_f_suffix(self):
        self.assertEqual(FLOAT32.format_literal(0.25), "0.25f")
        # General format: up to 8 significant digits, always ends with 'f'.
        s = FLOAT32.format_literal(1.0 / 3.0)
        self.assertTrue(s.endswith("f"))
        self.assertGreater(len(s), 3)  # at least "0.Xf"

    def test_c_typedef_comment(self):
        self.assertEqual(
            FLOAT32.c_typedef_comment(),
            "/* float32: IEEE 754 single precision */",
        )


# ================================================================ #
# Cross-check: singletons match the underlying classes              #
# ================================================================ #

class TestSingletons(unittest.TestCase):
    def test_ap_fixed_16_8_singleton(self):
        self.assertIsInstance(AP_FIXED_16_8, ApFixed)
        self.assertEqual(AP_FIXED_16_8.name, "ap_fixed<16,8>")
        self.assertEqual(AP_FIXED_16_8.bytes_per_elem, 2)

    def test_float32_singleton(self):
        self.assertIsInstance(FLOAT32, Float32)
        self.assertEqual(FLOAT32.bytes_per_elem, 4)


if __name__ == "__main__":
    unittest.main()


# ================================================================ #
# inference_buf_fill_float / read_float conversions (C, compiled)   #
# ================================================================ #

_CC = shutil.which("cc") or shutil.which("gcc")


@unittest.skipUnless(_CC, "no C compiler")
class TestBufFloatConversionsC(unittest.TestCase):
    """The generated buf_from_float / buf_to_float, compiled and run, match
    float_to_storage (clip + round half to even) and the storage decode."""

    def _run(self, dt, values):
        prog = (
            "#include <stdint.h>\n#include <stdio.h>\n"
            f"typedef {dt.c_type} Data_t;\n" + dt.c_buf_float_conversions() +
            "int main(void)\n{\n    float v;\n"
            "    while (fread(&v, sizeof v, 1, stdin) == 1) {\n"
            "        Data_t b = buf_from_float(v);\n"
            "        float  r = buf_to_float(b);\n"
            "        fwrite(&b, sizeof b, 1, stdout);\n"
            "        fwrite(&r, sizeof r, 1, stdout);\n"
            "    }\n    return 0;\n}\n")
        with tempfile.TemporaryDirectory() as td:
            src, exe = os.path.join(td, "t.c"), os.path.join(td, "t")
            with open(src, "w") as f:
                f.write(prog)
            subprocess.run([_CC, "-O2", "-Wall", "-Werror", "-o", exe, src], check=True)
            out = subprocess.run([exe], input=values.astype(np.float32).tobytes(),
                                 capture_output=True, check=True).stdout
        rec = np.dtype([("b", dt.np_storage), ("r", np.float32)])
        return np.frombuffer(out, dtype=rec)

    def test_fixed_point_matches_float_to_storage(self):
        rng = np.random.default_rng(3)
        for dt in (AP_FIXED_16_8, ApFixed(8, 4), ApFixed(32, 16)):
            with self.subTest(dtype=dt.name):
                s = 2.0 ** (dt.bytes_per_elem * 8 - dt._I)
                span = 2.0 ** (dt._I - 1)
                k = rng.integers(-int(span * s) - 50, int(span * s) + 50, 2000)
                values = np.concatenate([
                    rng.uniform(-1.5 * span, 1.5 * span, 4000),   # incl. saturation
                    (k + 0.5) / s,                                # exact ties
                    k / s,                                        # exact grid points
                    [0.0, -0.0, np.inf, -np.inf, span, -span]]).astype(np.float32)
                got = self._run(dt, values)
                want = dt.float_to_storage(values.astype(np.float64))
                np.testing.assert_array_equal(got["b"], want)
                signed = got["b"].view(dt._np_int).astype(np.float64)
                np.testing.assert_array_equal(got["r"], (signed / s).astype(np.float32))
                nan = self._run(dt, np.array([np.nan], np.float32))
                self.assertEqual(int(nan["b"][0]), 0)

    def test_float32_is_a_cast(self):
        values = np.array([0.1, -2.5, 1e30, -7.0], np.float32)
        got = self._run(FLOAT32, values)
        np.testing.assert_array_equal(got["b"], values)
        np.testing.assert_array_equal(got["r"], values)

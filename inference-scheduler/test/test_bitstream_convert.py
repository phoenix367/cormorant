# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.bitstream.convert — .bit header parser and bit→bin byte-swap."""

import os
import struct
import tempfile
import unittest
from pathlib import Path

from src.bitstream.convert import _parse_bit_header, bit_to_bin


# ---------------------------------------------------------------- #
# Minimal .bit byte-sequence builder                                #
# ---------------------------------------------------------------- #

def _make_bit(
    design:    bytes = b"design;Version=1.0\x00",
    part:      bytes = b"xck26-sfvc784-2LV-c\x00",
    date:      bytes = b"2026/05/12\x00",
    time:      bytes = b"19:00:00\x00",
    bitstream: bytes = b"\x11\x22\x33\x44\x55\x66\x77\x88",
    *,
    declared_bitstream_len: int = None,
) -> bytes:
    """Assemble a minimal Xilinx .bit-formatted byte sequence.

    declared_bitstream_len overrides the 0x65 length field — used to
    exercise the size-mismatch ValueError.
    """
    preamble = b"PREAMBLE\x00"
    out  = struct.pack(">h", len(preamble)) + preamble
    out += b"\x00\x01"                                      # unknown 2-byte field
    out += b"\x61" + struct.pack(">h", len(design)) + design
    out += b"\x62" + struct.pack(">h", len(part))   + part
    out += b"\x63" + struct.pack(">h", len(date))   + date
    out += b"\x64" + struct.pack(">h", len(time))   + time
    declared = declared_bitstream_len if declared_bitstream_len is not None else len(bitstream)
    out += b"\x65" + struct.pack(">i", declared) + bitstream
    return out


def _write_tmp_bit(payload: bytes) -> Path:
    fd, path = tempfile.mkstemp(suffix=".bit")
    os.close(fd)
    Path(path).write_bytes(payload)
    return Path(path)


# ---------------------------------------------------------------- #
# _parse_bit_header                                                 #
# ---------------------------------------------------------------- #

class TestParseBitHeader(unittest.TestCase):
    def test_happy_path_all_known_keys(self):
        bit    = _make_bit()
        parsed = _parse_bit_header(bit)
        # Null terminators are stripped by .rstrip("\x00").
        self.assertEqual(parsed["design"], "design;Version=1.0")
        self.assertEqual(parsed["part"],   "xck26-sfvc784-2LV-c")
        self.assertEqual(parsed["date"],   "2026/05/12")
        self.assertEqual(parsed["time"],   "19:00:00")
        self.assertEqual(parsed["data"],   b"\x11\x22\x33\x44\x55\x66\x77\x88")

    def test_bitstream_length_mismatch_raises(self):
        bit = _make_bit(declared_bitstream_len=999)
        with self.assertRaisesRegex(ValueError, "Bitstream length"):
            _parse_bit_header(bit)

    def test_unknown_header_field_raises(self):
        # Preamble: 2-byte length (=9) + 9-byte payload + 2-byte unknown = 13 bytes
        # before the first key byte.  Replace key 0x61 with 0x66 (unknown).
        bit     = bytearray(_make_bit())
        bit[13] = 0x66
        with self.assertRaisesRegex(ValueError, "Unknown .bit header field"):
            _parse_bit_header(bytes(bit))


# ---------------------------------------------------------------- #
# bit_to_bin — header strip + 32-bit big→little-endian byte swap    #
# ---------------------------------------------------------------- #

class TestBitToBin(unittest.TestCase):
    def test_byte_swaps_32bit_words(self):
        # Two 32-bit big-endian words → flipped per word.
        bs   = b"\x11\x22\x33\x44\x55\x66\x77\x88"
        path = _write_tmp_bit(_make_bit(bitstream=bs))
        try:
            result = bit_to_bin(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(result, b"\x44\x33\x22\x11\x88\x77\x66\x55")


if __name__ == "__main__":
    unittest.main()

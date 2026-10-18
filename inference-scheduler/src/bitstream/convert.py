# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Xilinx .bit → raw .bin conversion."""

import struct
from pathlib import Path

import numpy as np


def _parse_bit_header(data: bytes) -> dict:
    """Parse a Xilinx .bit file header; return metadata dict with a 'data' key."""
    offset = 0

    # First field: 2-byte length + payload (design name preamble)
    length = struct.unpack(">h", data[offset:offset + 2])[0]
    offset += 2 + length

    # Two-byte unknown field (usually 0x0001)
    offset += 2

    # Key–value fields until key 0x65 (the raw bitstream marker)
    result = {}
    while True:
        key = data[offset]
        offset += 1
        if key == 0x65:
            length = struct.unpack(">i", data[offset:offset + 4])[0]
            offset += 4
            if length + offset != len(data):
                raise ValueError("Bitstream length field does not match file size")
            result["data"] = data[offset:offset + length]
            break
        length = struct.unpack(">h", data[offset:offset + 2])[0]
        offset += 2
        value  = data[offset:offset + length].decode("ascii", errors="replace").rstrip("\x00")
        offset += length
        if key == 0x61:
            result["design"] = value
        elif key == 0x62:
            result["part"] = value
        elif key == 0x63:
            result["date"] = value
        elif key == 0x64:
            result["time"] = value
        else:
            raise ValueError(f"Unknown .bit header field: 0x{key:02x}")

    return result


def bit_to_bin(bit_path: Path) -> bytes:
    """
    Convert a Vivado .bit file to a flat .bin for the Linux fpga_manager.

    Strips the .bit header and byte-swaps 32-bit words from big-endian to
    little-endian, as required by the Zynq fpga_manager firmware interface.
    """
    parsed = _parse_bit_header(bit_path.read_bytes())
    return np.frombuffer(parsed["data"], dtype=">i4").byteswap().tobytes()

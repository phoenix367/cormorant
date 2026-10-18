# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Tests for src.bitstream.hwh — Vivado .hwh XML parsing."""

import os
import tempfile
import unittest
from pathlib import Path

from src.bitstream.hwh import parse_hwh_ps_params, parse_hwh_mem_topology


# ---------------------------------------------------------------- #
# HWH XML fixtures                                                  #
# ---------------------------------------------------------------- #

_PS_PARAMS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<EDKSYSTEM>
  <MODULES>
    <MODULE MODTYPE="zynq_ultra_ps_e" INSTANCE="ps_e_0">
      <PARAMETERS>
        <PARAMETER NAME="C_SAXIGP0_DATA_WIDTH" VALUE="128"/>
        <PARAMETER NAME="C_MAXIGP0_DATA_WIDTH" VALUE="32"/>
        <PARAMETER NAME="C_SAXIGP2_DATA_WIDTH" VALUE="128"/>
        <PARAMETER NAME="C_SOME_OTHER_PARAM"   VALUE="ignored"/>
        <PARAMETER NAME="C_NUM_F2P_INTR_INPUTS" VALUE="8"/>
      </PARAMETERS>
    </MODULE>
    <MODULE MODTYPE="axi_interconnect">
      <PARAMETERS>
        <PARAMETER NAME="C_SAXIGP0_DATA_WIDTH" VALUE="64"/>
      </PARAMETERS>
    </MODULE>
  </MODULES>
</EDKSYSTEM>
"""

_NO_PS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<EDKSYSTEM>
  <MODULES>
    <MODULE MODTYPE="axi_interconnect">
      <PARAMETERS/>
    </MODULE>
  </MODULES>
</EDKSYSTEM>
"""

# Five MEMRANGE entries exercise: dedupe vs PSDDR seed, fresh range,
# duplicate-skip, MEMTYPE filter, hex-parse-failure branch.
_MEM_TOPO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<EDKSYSTEM>
  <MEMRANGE MEMTYPE="MEMORY" BASEVALUE="0x00000000" HIGHVALUE="0x3FFFFFFF"/>
  <MEMRANGE MEMTYPE="MEMORY" BASEVALUE="0x80000000" HIGHVALUE="0xBFFFFFFF"/>
  <MEMRANGE MEMTYPE="MEMORY" BASEVALUE="0x00000000" HIGHVALUE="0x0FFFFFFF"/>
  <MEMRANGE MEMTYPE="OTHER"  BASEVALUE="0xC0000000" HIGHVALUE="0xFFFFFFFF"/>
  <MEMRANGE MEMTYPE="MEMORY" BASEVALUE="nothex"     HIGHVALUE="0x12345678"/>
</EDKSYSTEM>
"""


def _write_xml(xml: str) -> Path:
    fd, p = tempfile.mkstemp(suffix=".hwh")
    os.close(fd)
    path = Path(p)
    path.write_text(xml)
    return path


# ---------------------------------------------------------------- #
# parse_hwh_ps_params                                               #
# ---------------------------------------------------------------- #

class TestParseHwhPsParams(unittest.TestCase):
    def test_filters_only_axigp_data_width(self):
        path = _write_xml(_PS_PARAMS_XML)
        try:
            family, params = parse_hwh_ps_params(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(family, "zynq_ultra_ps_e")
        self.assertEqual(set(params), {
            "C_SAXIGP0_DATA_WIDTH",
            "C_MAXIGP0_DATA_WIDTH",
            "C_SAXIGP2_DATA_WIDTH",
        })
        self.assertEqual(params["C_SAXIGP0_DATA_WIDTH"], "128")
        self.assertEqual(params["C_MAXIGP0_DATA_WIDTH"], "32")
        self.assertEqual(params["C_SAXIGP2_DATA_WIDTH"], "128")
        # The non-AXIGP parameter and the second (non-PS) MODULE are ignored.
        self.assertNotIn("C_SOME_OTHER_PARAM",   params)
        self.assertNotIn("C_NUM_F2P_INTR_INPUTS", params)

    def test_raises_when_no_ps_module(self):
        path = _write_xml(_NO_PS_XML)
        try:
            with self.assertRaisesRegex(ValueError, "No PS IP module"):
                parse_hwh_ps_params(path)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------- #
# parse_hwh_mem_topology                                            #
# ---------------------------------------------------------------- #

class TestParseHwhMemTopology(unittest.TestCase):
    def test_psddr_seeded_and_unique_memory_ranges_appended(self):
        path = _write_xml(_MEM_TOPO_XML)
        try:
            topo = parse_hwh_mem_topology(path)
        finally:
            path.unlink(missing_ok=True)

        # PSDDR seed + 0x80000000 range; 0x00000000 dupes, OTHER, and the
        # nothex-base entry are all skipped.
        self.assertEqual(topo["m_count"], 2)

        psddr = topo["m_mem_data"][0]
        self.assertEqual(psddr["m_tag"],          "PSDDR")
        self.assertEqual(psddr["m_base_address"], 0)
        self.assertEqual(psddr["m_type"],         "MEM_DDR4")
        self.assertEqual(psddr["m_sizeKB"],       256 * 1024)

        mig = topo["m_mem_data"][1]
        self.assertEqual(mig["m_tag"],          "MIG1")
        self.assertEqual(mig["m_base_address"], 0x80000000)
        self.assertEqual(mig["m_sizeKB"],
                         (0xBFFFFFFF - 0x80000000 + 1) // 1024)


if __name__ == "__main__":
    unittest.main()

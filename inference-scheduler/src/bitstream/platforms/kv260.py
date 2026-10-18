# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

# KV260 / Zynq UltraScale+ platform constants for bitstream loading.
#
# Register tables and metadata are copied verbatim from PYNQ's
# pynq/pl_server/embedded_device.py.

# Minimal project XML required by xclbinutil to produce a MEM_TOPOLOGY-only
# xclbin.  Content beyond the board/part identifiers is irrelevant for our
# use; the file satisfies xclbinutil's mandatory EMBEDDED_METADATA section.
BLANK_METADATA = r"""<?xml version="1.0" encoding="UTF-8"?>
<project name="binary_container_1">
  <platform vendor="xilinx" boardid="kv260" name="name" featureRomTime="0">
    <version major="0" minor="1"/>
    <description/>
    <board name="xilinx.com:kv260:1.0" vendor="xilinx.com" fpga="xck26-sfvc784-2LV-c">
      <interfaces/>
      <memories><memory name="ddr4_0" type="ddr4" size="4GB"/></memories>
    </board>
    <build_flow/>
    <host architecture="unknown"/>
    <device name="fpga0" fpgaDevice="zynquplus:xck26:sfvc784:-2LV:c" addrWidth="0">
      <core name="OCL_REGION_0" target="bitstream" type="clc_region"
            clockFreq="0MHz" numComputeUnits="1">
        <kernelClocks>
          <clock port="KERNEL_CLK" frequency="100.000000MHz"/>
          <clock port="DATA_CLK"   frequency="100.000000MHz"/>
        </kernelClocks>
      </core>
    </device>
  </platform>
</project>
"""

# FPD SLCR — controls PS master AXI port widths (C_MAXIGP*)
FPD_SLCR_REG = {
    "C_MAXIGP0_DATA_WIDTH": [{"addr": 0xFD615000, "field": [9,  8]}],
    "C_MAXIGP1_DATA_WIDTH": [{"addr": 0xFD615000, "field": [11, 10]}],
    "C_MAXIGP2_DATA_WIDTH": [{"addr": 0xFF419000, "field": [9,  8]}],
}
FPD_SLCR_VALUE = {"32": 0, "64": 1, "128": 2}

# AXI Fabric Manager — controls PS slave AXI port widths (C_SAXIGP*)
AXIFM_REG = {
    "C_SAXIGP0_DATA_WIDTH": [
        {"addr": 0xFD360000, "field": [1, 0]},
        {"addr": 0xFD360014, "field": [1, 0]},
    ],
    "C_SAXIGP1_DATA_WIDTH": [
        {"addr": 0xFD370000, "field": [1, 0]},
        {"addr": 0xFD370014, "field": [1, 0]},
    ],
    "C_SAXIGP2_DATA_WIDTH": [
        {"addr": 0xFD380000, "field": [1, 0]},
        {"addr": 0xFD380014, "field": [1, 0]},
    ],
    "C_SAXIGP3_DATA_WIDTH": [
        {"addr": 0xFD390000, "field": [1, 0]},
        {"addr": 0xFD390014, "field": [1, 0]},
    ],
    "C_SAXIGP4_DATA_WIDTH": [
        {"addr": 0xFD3A0000, "field": [1, 0]},
        {"addr": 0xFD3A0014, "field": [1, 0]},
    ],
    "C_SAXIGP5_DATA_WIDTH": [
        {"addr": 0xFD3B0000, "field": [1, 0]},
        {"addr": 0xFD3B0014, "field": [1, 0]},
    ],
    "C_SAXIGP6_DATA_WIDTH": [
        {"addr": 0xFF9B0000, "field": [1, 0]},
        {"addr": 0xFF9B0014, "field": [1, 0]},
    ],
}
AXIFM_VALUE = {"32": 2, "64": 1, "128": 0}


def axi_port_width_writes(
    family: str, ps_params: dict
) -> list[tuple[int, int, int]]:
    """
    Return (addr, mask, value) RMW tuples for set_axi_port_width.

    Maps C_SAXIGP*/C_MAXIGP* DATA_WIDTH values from the HWH to the
    corresponding ZU+ FPD_SLCR / AXIFM register fields.  Returns an
    empty list for non-ZU+ families (processing_system7 etc.).
    """
    if family != "zynq_ultra_ps_e":
        return []

    writes: list[tuple[int, int, int]] = []
    for lut, val_table in [(FPD_SLCR_REG, FPD_SLCR_VALUE), (AXIFM_REG, AXIFM_VALUE)]:
        for param, regs in lut.items():
            width_str = ps_params.get(param)
            if width_str is None:
                continue
            val = val_table.get(width_str)
            if val is None:
                continue
            for reg in regs:
                hi, lo = reg["field"]
                mask = ((1 << (hi - lo + 1)) - 1) << lo
                writes.append((reg["addr"], mask, val << lo))
    return writes

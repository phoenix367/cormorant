"""Vivado hardware-handoff (.hwh) parsing."""

import xml.etree.ElementTree as ET
from pathlib import Path

_PS_MODTYPES = {"zynq_ultra_ps_e", "processing_system7"}


def parse_hwh_ps_params(hwh_path: Path) -> tuple[str, dict]:
    """
    Return (family, params) from the PS IP module in the HWH.

    family — "zynq_ultra_ps_e" or "processing_system7"
    params — {name: value} for all C_SAXIGP*/C_MAXIGP*_DATA_WIDTH parameters
    """
    root = ET.parse(hwh_path).getroot()
    for mod in root.iter("MODULE"):
        family = mod.get("MODTYPE", "")
        if family not in _PS_MODTYPES:
            continue
        params = {
            p.get("NAME"): p.get("VALUE")
            for p in mod.findall("./PARAMETERS/PARAMETER")
            if p.get("NAME", "").startswith(("C_SAXIGP", "C_MAXIGP"))
            and p.get("NAME", "").endswith("_DATA_WIDTH")
        }
        return family, params
    raise ValueError(
        f"No PS IP module (zynq_ultra_ps_e / processing_system7) found in {hwh_path}"
    )


def parse_hwh_clocks(hwh_path: Path) -> dict:
    """
    Return {"pl0_mhz", "kernel_mhz"} from the HWH.

    pl0_mhz    — the PS's PL0 clock the design expects (the PS module's
                 PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ), None if absent
    kernel_mhz — the kernels' ap_clk (CLKFREQUENCY of the first ap_clk
                 input found), None if absent.  With the MMCM design
                 (clk_wiz_0, doc/plans/FMAX_250_PLAN.md) it differs from PL0.
    """
    root = ET.parse(hwh_path).getroot()
    pl0 = kernel = None
    for mod in root.iter("MODULE"):
        if mod.get("MODTYPE", "") in _PS_MODTYPES and pl0 is None:
            for p in mod.findall("./PARAMETERS/PARAMETER"):
                if p.get("NAME") == "PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ":
                    pl0 = float(p.get("VALUE"))
        if kernel is None:
            for port in mod.iter("PORT"):
                if port.get("NAME") == "ap_clk" and port.get("DIR") == "I" \
                        and port.get("CLKFREQUENCY"):
                    kernel = int(port.get("CLKFREQUENCY")) / 1e6
                    break
    return {"pl0_mhz": pl0, "kernel_mhz": kernel}


def parse_hwh_mem_topology(hwh_path: Path) -> dict:
    """
    Build the MEM_TOPOLOGY dict for xclbinutil from the HWH.

    Always seeds with a PSDDR bank at index 0 (address 0, 256 MiB) so that
    xclAllocBO(..., flags=0) in inference_buf.c resolves to a named bank.
    Additional MEMTYPE=MEMORY ranges from the HWH are appended.
    """
    root = ET.parse(hwh_path).getroot()

    mem_data: list[dict] = [{
        "m_type":         "MEM_DDR4",
        "m_used":         1,
        "m_sizeKB":       256 * 1024,
        "m_tag":          "PSDDR",
        "m_base_address": 0,
    }]

    seen: set[int] = {0}
    for mr in root.iter("MEMRANGE"):
        if mr.get("MEMTYPE") != "MEMORY":
            continue
        try:
            base = int(mr.get("BASEVALUE", "0"), 16)
            high = int(mr.get("HIGHVALUE", "0"), 16)
        except ValueError:
            continue
        if base in seen:
            continue
        seen.add(base)
        mem_data.append({
            "m_type":         "MEM_DDR4",
            "m_used":         1,
            "m_sizeKB":       max((high - base + 1) // 1024, 1),
            "m_tag":          f"MIG{len(mem_data)}",
            "m_base_address": base,
        })

    return {"m_count": len(mem_data), "m_mem_data": mem_data}

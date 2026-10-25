# Copyright 2026 Ivan Gubochkin
# SPDX-License-Identifier: Apache-2.0

"""A bitstream load the board can replay by itself.

upload_bitstream.py loads a bitstream from the host (loader.py).  After a
reboot the board runs only the starter-kit design, so a service that starts
on the board (the chat server's systemd unit, demo/chat/deploy.py) has to
load its bitstream itself: demo/chat/fpga_load.py does upload_bitstream.py's
board steps from a bundle of four files, which ``make_bundle`` builds on the
host from the same inputs as upload_bitstream.py:

  bitstream.bin   the flat .bin (bit_to_bin)
  overlay.dtbo    the device-tree overlay
  design.xclbin   the xclbin with the HWH's MEM_TOPOLOGY (needs xclbinutil; loaded where
                  upload_bitstream.py loads it, before the overlay, where zocl has no
                  device yet, so in practice not: the buffers come from CMA)
  load.json       {id, overlay, pl0_hz, axi_writes: [[addr, mask, value]],
                   uio_nodes: ["name@addr"], uio_irqs: {"name@addr": GIC id}}
"""

from __future__ import annotations

import json
from pathlib import Path

from ..perf_calls import bitstream_id_of_bin
from .convert import bit_to_bin
from .hwh import parse_hwh_clocks, parse_hwh_ps_params
from .loader import dtbo_uio_irqs, dtbo_uio_nodes
from .platforms import kv260
from .xclbin import build_xclbin

BUNDLE_FILES = ("bitstream.bin", "overlay.dtbo", "design.xclbin", "load.json")


def resolve_bitstream_config(config: Path) -> dict:
    """bit / hwh / dtbo / overlay_name / xclbinutil of a bitstream config
    (bitstream_config_kv260.json), resolved as upload_bitstream.py does:
    paths relative to the config's directory, the hwh next to the .bit when
    unset, the overlay named after the .dtbo."""
    config = Path(config).expanduser().resolve()
    bs = json.loads(config.read_text()).get("bitstream") or {}

    def path(key):
        v = bs.get(key)
        if not v:
            return None
        p = Path(v).expanduser()
        return (p if p.is_absolute() else config.parent / p).resolve()

    bit, dtbo = path("bit"), path("dtbo")
    if bit is None or dtbo is None:
        raise ValueError(f"{config}: bitstream.bit and bitstream.dtbo are required")
    hwh = path("hwh") or bit.with_suffix(".hwh")
    for p in (bit, hwh, dtbo):
        if not p.exists():
            raise FileNotFoundError(f"{config}: {p} not found")
    return {"bit": bit, "hwh": hwh, "dtbo": dtbo,
            "overlay_name": bs.get("overlay_name") or dtbo.stem,
            "xclbinutil": bs.get("xclbinutil") or "xclbinutil"}


def make_bundle(bit: Path, hwh: Path, dtbo: Path, overlay_name: str,
                xclbinutil: str = "xclbinutil") -> dict[str, bytes]:
    """{file name: content} of the bundle (BUNDLE_FILES)."""
    data = bit_to_bin(Path(bit))
    family, ps_params = parse_hwh_ps_params(Path(hwh))
    pl0_mhz = parse_hwh_clocks(Path(hwh))["pl0_mhz"]
    nodes = dtbo_uio_nodes(dtbo)
    if not nodes:
        raise RuntimeError(f"{dtbo}: no generic-uio node found (is dtc installed?)")
    load = {"id": bitstream_id_of_bin(data), "overlay": overlay_name,
            "pl0_hz": None if pl0_mhz is None else round(pl0_mhz * 1e6),
            "axi_writes": [list(w) for w in kv260.axi_port_width_writes(family, ps_params)],
            "uio_nodes": nodes, "uio_irqs": dtbo_uio_irqs(dtbo)}
    return {"bitstream.bin": data, "overlay.dtbo": Path(dtbo).read_bytes(),
            "design.xclbin": build_xclbin(Path(hwh), kv260.BLANK_METADATA, xclbinutil),
            "load.json": (json.dumps(load, indent=1) + "\n").encode()}

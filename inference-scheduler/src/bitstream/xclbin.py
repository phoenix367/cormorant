# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""Build a minimal xclbin containing only a MEM_TOPOLOGY section."""

import json
import subprocess
import tempfile
from pathlib import Path

from .hwh import parse_hwh_mem_topology


def build_xclbin(
    hwh_path: Path,
    blank_metadata: str,
    xclbinutil: str = "xclbinutil",
) -> bytes:
    """
    Create a minimal xclbin with a MEM_TOPOLOGY section derived from the HWH.

    The xclbin is loaded into the zocl DRM driver so that xclAllocBO resolves
    memory bank 0 to the named PSDDR bank instead of falling back to CMA with
    "Allocating BO from CMA for invalid or unused memory index[0]" warnings.

    blank_metadata — platform-specific EMBEDDED_METADATA XML (e.g. kv260.BLANK_METADATA)
    xclbinutil     — command name or path; must be on PATH or given explicitly
    """
    mem_topology = parse_hwh_mem_topology(hwh_path)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "metadata.xml").write_text(blank_metadata)
        (td_path / "mem.json").write_text(json.dumps({"mem_topology": mem_topology}))
        result = subprocess.run(
            [
                xclbinutil,
                "--add-section=EMBEDDED_METADATA:RAW:metadata.xml",
                "--add-section=MEM_TOPOLOGY:JSON:mem.json",
                "--output", "design.xclbin",
                "--skip-bank-grouping",
            ],
            cwd=td_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xclbinutil failed (rc={result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return (td_path / "design.xclbin").read_bytes()

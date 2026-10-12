"""
RowOpsKernel capability resolved from the platform JSON.

``unit`` says whether the platform's bitstream carries the row-ops unit
(doc/plans/HOSTOPS_PLAN.md, doc/kernels/ROWOPS_RTL_KERNEL.md): the fifth
kernel lane that runs the transformer's row ops — LayerNorm / RMSNorm, the
residual add, SiLU · up, dequantisation, RoPE, transposes and copies — in
fixed point.  Where it is false the scheduler keeps those ops on the host
(``src/rowops_nodes.py``: the kernel nodes are chosen only with the unit).

JSON shape (only the fields this module reads)::

    {
      "kernels": {
        "rowops": {
          "unit": false
        }
      }
    }

Platform selection as for the other kernels (``_matmul_hw_config``): the
``AXI_PLATFORM`` environment variable picks ``platforms/<name>.json``,
default ``kv260``.  ``AXI_ROWOPS`` (``0`` / ``1``) overrides the JSON for
the scheduler — ``1`` to generate for a bitstream with the unit before the
platform file says so, ``0`` for one without it.  A missing or malformed
field raises ``RowopsHwConfigError``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

_SCHEDULER_DIR = Path(__file__).resolve().parent.parent
_PLATFORMS_DIR = _SCHEDULER_DIR.parent / "platforms"
_DEFAULT_PLATFORM = "kv260"


class RowopsHwConfigError(RuntimeError):
    """Platform JSON is missing / malformed for the rowops kernel."""


def resolve(platform_name: str = None) -> Dict[str, object]:
    """``{"ROWOPS_UNIT": bool}`` for the given platform (``None``:
    ``AXI_PLATFORM``, else ``kv260``), ``AXI_ROWOPS`` applied."""
    if platform_name is None:
        platform_name = os.environ.get("AXI_PLATFORM", _DEFAULT_PLATFORM)
    path = _PLATFORMS_DIR / f"{platform_name}.json"
    if not path.is_file():
        raise RowopsHwConfigError(
            f"Platform file not found: {path}.  Set AXI_PLATFORM to a "
            f"<name> for which platforms/<name>.json exists.")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise RowopsHwConfigError(f"Platform JSON {path} is not valid JSON: {e}") from e
    try:
        v = data["kernels"]["rowops"]["unit"]
    except (KeyError, TypeError) as e:
        raise RowopsHwConfigError(
            f"Platform JSON {path}: 'kernels.rowops.unit' is required (true when the "
            f"bitstream carries the row-ops unit).") from e
    if not isinstance(v, bool):
        raise RowopsHwConfigError(
            f"Platform JSON {path}: 'kernels.rowops.unit' must be true or false, got {v!r}.")
    env = os.environ.get("AXI_ROWOPS")
    if env:
        if env not in ("0", "1"):
            raise RowopsHwConfigError(f"AXI_ROWOPS must be 0 or 1, got {env!r}.")
        v = env == "1"
    return {"ROWOPS_UNIT": v}


_CFG = resolve()

# Exported: whether the platform's bitstream has the row-ops unit.
ROWOPS_UNIT: bool = _CFG["ROWOPS_UNIT"]

__all__ = ("ROWOPS_UNIT", "RowopsHwConfigError", "resolve")
